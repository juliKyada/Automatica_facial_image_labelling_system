import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import { writeFile, unlink } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'

const execAsync = promisify(exec)

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { image } = body

    if (!image) {
      return NextResponse.json(
        { success: false, error: 'No image provided' },
        { status: 400 }
      )
    }

    // For local development: Run Python script
    // For Vercel: The vercel.json routes /api/predict to /api/predict.py automatically
    if (process.env.VERCEL) {
      // On Vercel, the Python function will be called via the routing in vercel.json
      // But Next.js API routes take precedence, so we need to proxy or skip this
      // Actually, let's just use the Python script for local dev
    }

    // For local development: Run Python script
    try {
      // Create a temporary file with the image data
      const tempFile = join(tmpdir(), `predict_${Date.now()}.json`)
      await writeFile(tempFile, JSON.stringify({ image }))

      // Get the project root
      const projectRoot = process.cwd()
      const pythonScript = join(projectRoot, 'api', 'predict_local.py')

      // Try to find Python with tensorflow installed
      // First, try to use the same Python that would be used for the Streamlit app
      // Check if there's a virtual environment or use system Python
      let pythonCmd = 'python'
      let pythonPath = ''
      
      // Try different Python commands in order of preference
      const pythonCommands = ['python', 'python3', 'py']
      
      for (const cmd of pythonCommands) {
        try {
          // Check if this Python has tensorflow
          const { stdout: versionOut } = await execAsync(`${cmd} --version`)
          // Try importing tensorflow to see if it's available
          const { stdout: tfCheck } = await execAsync(
            `${cmd} -c "import tensorflow; print('ok')" 2>&1`,
            { timeout: 5000 }
          )
          if (tfCheck.includes('ok')) {
            pythonCmd = cmd
            break
          }
        } catch (e) {
          // Try next command
          continue
        }
      }

      // Run Python script
      const { stdout, stderr } = await execAsync(
        `${pythonCmd} "${pythonScript}" "${tempFile}"`,
        { 
          cwd: projectRoot, 
          maxBuffer: 10 * 1024 * 1024,
          env: { ...process.env, PYTHONUNBUFFERED: '1' }
        }
      )

      // Clean up temp file
      await unlink(tempFile).catch(() => {})

      if (stderr && !stdout) {
        throw new Error(stderr)
      }

      const result = JSON.parse(stdout || '{}')
      return NextResponse.json(result)
    } catch (pythonError: any) {
      console.error('Python execution error:', pythonError)
      return NextResponse.json(
        { 
          success: false, 
          error: `Python execution failed: ${pythonError.message}. Make sure Python and dependencies (tensorflow, opencv-python, numpy, Pillow) are installed.` 
        },
        { status: 500 }
      )
    }
  } catch (error: any) {
    console.error('API route error:', error)
    return NextResponse.json(
      { success: false, error: error.message || 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}

