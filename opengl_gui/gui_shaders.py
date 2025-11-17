from OpenGL.GL import *
from OpenGL.GL import shaders

class ShaderProgram():

    def __init__(self):
        self.vs = None
        self.fs = None

        self.uniform_functions = {}

    def compile(self):
        self.shader_program = shaders.compileProgram(shaders.compileShader(self.vs, GL_VERTEX_SHADER), shaders.compileShader(self.fs, GL_FRAGMENT_SHADER))

        return self

class DefaultShader(ShaderProgram):

    def __init__(self):
        super(DefaultShader, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform vec2 properties;

            layout (location = 0) in vec2 position;
            out float highlight;
            void main()
            {
                highlight = properties[1];
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x, tmp.y, properties[0], tmp.z);
            }
        """

        self.fs = \
        """
            #version 450 
            uniform vec4 color;
            in float highlight;

            out vec4 fragColor;

            void main()
            {
                fragColor = color;
                fragColor[0] = clamp(fragColor[0]*highlight, 0.0, 1.0);
                fragColor[1] = clamp(fragColor[1]*highlight, 0.0, 1.0);
                fragColor[2] = clamp(fragColor[2]*highlight, 0.0, 1.0);
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location = glGetUniformLocation(self.shader_program, "transform")
        self.color_uniform_location = glGetUniformLocation(self.shader_program, "color")
        self.properties_uniform_location = glGetUniformLocation(self.shader_program, "properties")
        

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["color"]    = lambda C: glUniform4fv(self.color_uniform_location, 1, C)
        self.uniform_functions["properties"] = lambda P: glUniform2fv(self.properties_uniform_location, 1, P)


        return self

class TextShader(ShaderProgram):

    def __init__(self):
        super(TextShader, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform float depth;

            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoordIn;
            out vec2 texCoordOut;
            void main()
            {
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,depth,tmp.z);
                texCoordOut = texCoordIn;
            }
        """

        self.fs = \
        """
            #version 450 
            uniform vec4 color;
            uniform sampler2D text;

            in vec2 texCoordOut;
            out vec4 fragColor;

            void main()
            {
                fragColor = vec4(vec3(color), texture(text, texCoordOut).r);
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location = glGetUniformLocation(self.shader_program, "transform")
        self.color_uniform_location = glGetUniformLocation(self.shader_program, "color")
        self.depth_uniform_location  = glGetUniformLocation(self.shader_program, "depth")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["color"]    = lambda C: glUniform4fv(self.color_uniform_location, 1, C)
        self.uniform_functions["depth"]    = lambda D: glUniform1f(self.depth_uniform_location, D)

        return self

class TextureShaderBGR(ShaderProgram):

    def __init__(self):
        super(TextureShaderBGR, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform vec3 properties;

            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoordIn;
            out vec2 texCoordOut;
            out float alpha;
            void main()
            {
                alpha = properties[1];
                float zoom = properties[2];
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,properties[0],tmp.z);
                texCoordOut = (texCoordIn + (zoom - 1)/2)/zoom;
            }
        """

        self.fs = \
        """
            #version 450 
            uniform sampler2D text;
            in float alpha;

            in vec2 texCoordOut;
            out vec4 fragColor;

            void main()
            {
                vec3 tmp = vec3(texture(text, texCoordOut));

                fragColor = vec4(tmp, alpha);
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location  = glGetUniformLocation(self.shader_program, "transform")
        self.properties_uniform_location = glGetUniformLocation(self.shader_program, "properties")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["properties"] = lambda P: glUniform3fv(self.properties_uniform_location, 1, P)
  

        return self

class TextureShaderRGBA(ShaderProgram):

    def __init__(self):
        super(TextureShaderRGBA, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform vec2 properties;

            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoordIn;
            out vec2 texCoordOut;
            void main()
            {
                float zoom = properties[1];
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,properties[0],tmp.z);
                texCoordOut = (texCoordIn + (zoom - 1)/2)/zoom;
            }
        """

        self.fs = \
        """
            #version 450 
            uniform sampler2D text;

            in vec2 texCoordOut;
            out vec4 fragColor;

            void main()
            {
                fragColor = vec4(texture(text, texCoordOut));
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location  = glGetUniformLocation(self.shader_program, "transform")
        self.properties_uniform_location = glGetUniformLocation(self.shader_program, "properties")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["properties"] = lambda P: glUniform2fv(self.properties_uniform_location, 1, P)
  
        return self

class TextureShaderR(ShaderProgram):

    def __init__(self):
        super(TextureShaderR, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform float depth;
            uniform float zoom;

            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoordIn;
            out vec2 texCoordOut;
            void main()
            {
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,depth,tmp.z);
                texCoordOut = (texCoordIn + (zoom - 1)/2)/zoom;
            }
        """

        self.fs = \
        """
            #version 450 
            uniform sampler2D text;
            uniform vec4 color;

            in vec2 texCoordOut;
            out vec4 fragColor;

            void main()
            {
                vec4 cl = vec4(texture(text, texCoordOut));
                
                fragColor = vec4(vec3(color), cl[0]*color[3]);
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location = glGetUniformLocation(self.shader_program, "transform")
        self.color_uniform_location = glGetUniformLocation(self.shader_program, "color")
        self.depth_uniform_location  = glGetUniformLocation(self.shader_program, "depth")
        self.zoom_uniform_location = glGetUniformLocation(self.shader_program, "zoom")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["color"]    = lambda C: glUniform4fv(self.color_uniform_location, 1, C)
        self.uniform_functions["depth"]     = lambda D: glUniform1f(self.depth_uniform_location, D)
        self.uniform_functions["zoom"] = lambda Z: glUniform1f(self.zoom_uniform_location, Z)

        return self

class CircleShader(DefaultShader):

    def __init__(self):
        super(CircleShader, self).__init__()

        self.vs = \
        """
            #version 450 
            uniform mat3 transform;
            uniform vec2 properties;

            layout (location = 0) in vec2 position;
            out vec2 fragPosition;
            out float highlight;

            void main()
            {
                highlight = properties[1];

                fragPosition = position;
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,properties[0],tmp.z);
            }
        """

        self.fs = \
        """
            #version 450 
            uniform vec4 color;
            in float highlight;

            in vec2 fragPosition;
            out vec4 fragColor;

            void main()
            {
                float x = 2 * fragPosition[0] - 1;
                float y = 2 * fragPosition[1] - 1;

                float inside = x*x + y*y;
                float alpha0 = 1.0f - smoothstep(1.0 - fwidth(inside), 1.0, inside);
                //float alpha1 = smoothstep(0.8 - fwidth(inside), 0.8, inside);

                fragColor = color;
                fragColor[0] = clamp(fragColor[0]*highlight, 0.0, 1.0);
                fragColor[1] = clamp(fragColor[1]*highlight, 0.0, 1.0);
                fragColor[2] = clamp(fragColor[2]*highlight, 0.0, 1.0);
                fragColor[3] = alpha0;
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location  = glGetUniformLocation(self.shader_program, "transform")
        self.color_uniform_location     = glGetUniformLocation(self.shader_program, "color")
        self.properties_uniform_location = glGetUniformLocation(self.shader_program, "properties")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["color"]    = lambda C: glUniform4fv(self.color_uniform_location, 1, C)
        self.uniform_functions["properties"] = lambda P: glUniform2fv(self.properties_uniform_location, 1, P)

        return self

class PortalShader(ShaderProgram):

    def __init__(self):
        super(PortalShader, self).__init__()

        self.vs = \
        """
            #version 450
            uniform mat3 transform;
            uniform vec4 properties;

            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoordIn;

            out vec2 texCoord;
            out vec2 fragmentPosition;

            void main()
            {
                // Subtract 0.5 to account for [0,1] vertices
                fragmentPosition = position - 0.5;
                texCoord = texCoordIn;
                vec3 tmp = transform*vec3(position,1.0);
                gl_Position = vec4(tmp.x,tmp.y,properties[0],tmp.z);
            }
        """

        self.fs = \
        """
            #version 450
            uniform int part;
            uniform vec4 properties;
            uniform vec4 color;

            in vec2 fragmentPosition;
            in vec2 texCoord;
            out vec4 fragmentColor;

            // Uniform to named variable
            float depth     = properties[0];
            float aspect    = properties[1];
            float radius    = properties[2];
            float thickness = properties[3];

            void main()
            {
                float d = length(vec2(fragmentPosition[0]*aspect, fragmentPosition[1]));

                if (part == 0 && d < radius ||
                    part == 1 && (d <= radius+thickness) && (d >= radius))
                {
                    fragmentColor = color;
                }
                else
                {
                    discard;
                }
            }
        """

    def generate_uniform_functions(self):

        self.transform_uniform_location = glGetUniformLocation(self.shader_program, "transform")
        self.part_uniform_location   = glGetUniformLocation(self.shader_program, "part")
        self.properties_uniform_location = glGetUniformLocation(self.shader_program, "properties")
        self.color_uniform_location = glGetUniformLocation(self.shader_program, "color")

        self.uniform_functions["transform"] = lambda M: glUniformMatrix3fv(self.transform_uniform_location, 1, GL_TRUE, M)
        self.uniform_functions["part"]     = lambda P: glUniform1i(self.part_uniform_location, P)
        self.uniform_functions["properties"] = lambda P: glUniform4fv(self.properties_uniform_location, 1, P)
        self.uniform_functions["color"] = lambda C: glUniform4fv(self.color_uniform_location, 1, C)

        return self
